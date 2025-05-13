export interface AgendaItem {
  id: string;
  headline: string;
  time: number; // in minutes
}

export interface Preset {
  name: string;
  agenda: AgendaItem[];
}
