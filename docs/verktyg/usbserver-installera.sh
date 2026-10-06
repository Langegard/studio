#!/usr/bin/env bash
# usbserver-installera.sh — gör denna maskin till USB/IP-server (usbip).
# Tänkt för en Linux-nod som alltid är på. Körs som root. Kan köras om utan skada.
#
# Användning:
#   sudo ./usbserver-installera.sh [--lan 192.168.1.0/24]
#
#   --lan  Hemnätets adressintervall. Port 3240 öppnas då för det intervallet
#          och för tailscale0. Utan --lan öppnas den bara för tailscale0.
#
# Efteråt väljer du vad som lånas ut med kommandot `usbdela` (se slutet).
# Lagringsenheter och hubbar vägrar usbdela alltid att låna ut (bandregeln:
# diskar delas som nätverksmappar, och aldrig pappas disk).
set -euo pipefail

LAN_CIDR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lan) LAN_CIDR="${2:?--lan kräver ett intervall, t.ex. 192.168.1.0/24}"; shift 2 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "Okänt argument: $1" >&2; exit 2 ;;
  esac
done

[[ $EUID -eq 0 ]] || { echo "Kör med sudo." >&2; exit 1; }
[[ "$(uname -s)" == Linux ]] || { echo "Fungerar bara på Linux." >&2; exit 1; }
command -v systemctl >/dev/null || { echo "Kräver systemd." >&2; exit 1; }

echo "== 1/5 Installerar usbip-verktygen"
if command -v apt-get >/dev/null; then
  apt-get update -qq
  if apt-cache show usbip >/dev/null 2>&1; then
    apt-get install -y usbip                                         # Debian
  else
    apt-get install -y linux-tools-generic "linux-tools-$(uname -r)" # Ubuntu
  fi
elif command -v dnf >/dev/null; then
  dnf install -y usbip
elif command -v pacman >/dev/null; then
  pacman -S --needed --noconfirm usbip
else
  echo "Okänd pakethanterare. Installera usbip för hand och kör om." >&2; exit 1
fi

hitta() {
  local p
  p="$(command -v "$1" 2>/dev/null || true)"
  [[ -z $p && -x /usr/lib/linux-tools/$(uname -r)/$1 ]] && p="/usr/lib/linux-tools/$(uname -r)/$1"
  echo "$p"
}
USBIP="$(hitta usbip)"
USBIPD="$(hitta usbipd)"
[[ -n $USBIP && -n $USBIPD ]] || { echo "Hittar inte usbip/usbipd efter installationen." >&2; exit 1; }
SYSTEMCTL="$(command -v systemctl)"

echo "== 2/5 Kärnmodulen usbip_host laddas nu och vid varje uppstart"
echo usbip_host > /etc/modules-load.d/usbip.conf
modprobe usbip_host

echo "== 3/5 Installerar hjälpkommandot usbdela"
mkdir -p /etc/usbdela
[[ -f /etc/usbdela/enheter ]] || cat > /etc/usbdela/enheter <<'EOF'
# Enheter som lånas ut över nätverket, en per rad som leverantör:produkt.
# Redigera med `usbdela dela ...` och `usbdela sluta ...` hellre än för hand.
EOF

cat > /usr/local/sbin/usbdela <<'EOF'
#!/usr/bin/env bash
# usbdela — välj vilka USB-enheter servern lånar ut över nätverket.
#   usbdela lista              alla enheter i servern, och vad som lånas ut
#   usbdela dela VVVV:PPPP     låna ut en enhet (sparas till nästa uppstart)
#   usbdela sluta VVVV:PPPP    sluta låna ut den
#   usbdela aterbind           binder om allt i listan (körs automatiskt)
set -euo pipefail
USBIP=@USBIP@
LISTA=/etc/usbdela/enheter

busids() {  # alla busid som har detta leverantör:produkt
  "$USBIP" list -p -l | sed -n "s/^busid=\([^#]*\)#usbid=$1#.*$/\1/p"
}
utlanad() { [[ "$(readlink -f "/sys/bus/usb/devices/$1/driver" 2>/dev/null)" == */usbip-host ]]; }
sparrad() {  # skriver ut skälet och lyckas om enheten aldrig får lånas ut
  local d=/sys/bus/usb/devices/$1 c
  [[ "$(cat "$d/bDeviceClass")" == 09 ]] && { echo "hubb"; return 0; }
  for c in "$d":*/bInterfaceClass; do
    [[ -e $c ]] || continue
    case "$(cat "$c")" in
      08) echo "lagringsenhet"; return 0 ;;
      09) echo "hubb"; return 0 ;;
    esac
  done
  return 1
}
har_hid() { grep -qx 03 "/sys/bus/usb/devices/$1":*/bInterfaceClass 2>/dev/null; }
giltig() { [[ $1 =~ ^[0-9a-f]{4}:[0-9a-f]{4}$ ]] || { echo "Ange som vvvv:pppp, t.ex. 046d:085e (små bokstäver)." >&2; exit 2; }; }
bind_en() {
  local b=$1 skal
  utlanad "$b" && return 0
  if skal="$(sparrad "$b")"; then echo "Vägrar $b: $skal." >&2; return 1; fi
  "$USBIP" bind -b "$b" && echo "Lånas ut: $b"
}

[[ $EUID -eq 0 ]] || { echo "Kör med sudo." >&2; exit 1; }
case "${1:-lista}" in
  lista)
    "$USBIP" list -l
    echo "-- Lånas ut just nu:"
    for d in /sys/bus/usb/drivers/usbip-host/*-*; do [[ -e $d ]] && echo "   ${d##*/}"; done
    echo "-- Sparad lista ($LISTA):"
    grep -v '^#' "$LISTA" | sed 's/^/   /' || true
    ;;
  dela)
    giltig "${2:-}"; v=$2
    mapfile -t bs < <(busids "$v")
    [[ ${#bs[@]} -gt 0 ]] || { echo "Ingen enhet $v är ansluten." >&2; exit 1; }
    for b in "${bs[@]}"; do
      if skal="$(sparrad "$b")"; then echo "Vägrar: $v är en $skal. Den lånas aldrig ut." >&2; exit 1; fi
      har_hid "$b" && echo "OBS: $v är mus/tangentbord-typ. Servern själv tappar den medan den lånas ut."
    done
    grep -qx "$v" "$LISTA" || echo "$v" >> "$LISTA"
    for b in "${bs[@]}"; do bind_en "$b" || true; done
    ;;
  sluta)
    giltig "${2:-}"; v=$2
    sed -i "/^$v\$/d" "$LISTA"
    while read -r b; do
      [[ -n $b ]] && utlanad "$b" && "$USBIP" unbind -b "$b" && echo "Tillbaka hos servern: $b"
    done < <(busids "$v")
    ;;
  aterbind)
    sleep 1   # låt en nyss inkopplad enhet hinna dyka upp
    while read -r v; do
      [[ -z $v || $v == \#* ]] && continue
      while read -r b; do [[ -n $b ]] && { bind_en "$b" || true; }; done < <(busids "$v")
    done < "$LISTA"
    ;;
  *) sed -n '2,7p' "$0"; exit 2 ;;
esac
EOF
sed -i "s|@USBIP@|$USBIP|" /usr/local/sbin/usbdela
chmod 755 /usr/local/sbin/usbdela

echo "== 4/5 Tjänster: usbipd vid uppstart, och ombindning när något kopplas in"
cat > /etc/systemd/system/usbdela-usbipd.service <<EOF
[Unit]
Description=USB/IP-server (usbipd) för usbdela
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStartPre=/sbin/modprobe usbip_host
ExecStart=$USBIPD
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/usbdela-aterbind.service <<'EOF'
[Unit]
Description=Lånar ut enheterna i /etc/usbdela/enheter igen
Requires=usbdela-usbipd.service
After=usbdela-usbipd.service

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/usbdela aterbind

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/udev/rules.d/90-usbdela.rules <<EOF
# Kopplas en enhet ur och i igen tappar den sin utlåning; detta lånar ut den igen.
ACTION=="add", SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", RUN+="$SYSTEMCTL --no-block start usbdela-aterbind.service"
EOF

systemctl daemon-reload
udevadm control --reload
systemctl enable --now usbdela-usbipd.service
systemctl enable usbdela-aterbind.service

echo "== 5/5 Brandvägg för port 3240"
if command -v ufw >/dev/null && ufw status | grep -q "Status: active"; then
  ufw allow in on tailscale0 to any port 3240 proto tcp
  [[ -n $LAN_CIDR ]] && ufw allow from "$LAN_CIDR" to any port 3240 proto tcp
  echo "ufw: port 3240 öppen för tailscale0${LAN_CIDR:+ och $LAN_CIDR}."
else
  cat <<'EOF'
VARNING: ingen aktiv ufw. Port 3240 nås nu av alla som når maskinen, och
usbip har varken lösenord eller kryptering. Sätt upp brandväggen innan du
lånar ut något, eller säg till så hjälper jag dig.
EOF
fi

cat <<EOF

Klart. Så här fortsätter du:
  sudo usbdela lista                 se enheterna och deras vvvv:pppp
  sudo usbdela dela 046d:085e        låna ut en enhet
  sudo usbdela sluta 046d:085e       ta tillbaka den

Från en Linux-klient:
  sudo modprobe vhci-hcd
  usbip list -r $(hostname -s)
  sudo usbip attach -r $(hostname -s) -b <busid>

Från Windows: installera usbip-win2 och kör
  usbip.exe attach -r $(hostname -s) -b <busid>
EOF
