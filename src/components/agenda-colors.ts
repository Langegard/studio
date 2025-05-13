// These HSL strings correspond to the CSS variables defined in globals.css
export const AGENDA_ITEM_COLORS: string[] = [
  'hsl(var(--agenda-color-1))',
  'hsl(var(--agenda-color-2))',
  'hsl(var(--agenda-color-3))',
  'hsl(var(--agenda-color-4))',
  'hsl(var(--agenda-color-5))',
  'hsl(var(--agenda-color-6))',
];

export const getAgendaItemColor = (index: number): string => {
  return AGENDA_ITEM_COLORS[index % AGENDA_ITEM_COLORS.length];
};
