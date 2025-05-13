// 'use server'
'use server';
/**
 * @fileOverview An AI agent to assist with time management during meetings.
 *
 * - timeManagementAssistant - A function that provides real-time alerts and suggestions for staying on track with the meeting agenda.
 * - TimeManagementAssistantInput - The input type for the timeManagementAssistant function.
 * - TimeManagementAssistantOutput - The return type for the timeManagementAssistant function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const TimeManagementAssistantInputSchema = z.object({
  agendaItem: z.string().describe('The current agenda item.'),
  timeAllocated: z.number().describe('The time allocated for the current agenda item in minutes.'),
  timeElapsed: z.number().describe('The time elapsed for the current agenda item in minutes.'),
});
export type TimeManagementAssistantInput = z.infer<typeof TimeManagementAssistantInputSchema>;

const TimeManagementAssistantOutputSchema = z.object({
  isOverTime: z.boolean().describe('Whether the agenda item is running over the allocated time.'),
  suggestions: z.array(z.string()).describe('Suggestions on how to get back on track.'),
});
export type TimeManagementAssistantOutput = z.infer<typeof TimeManagementAssistantOutputSchema>;

export async function timeManagementAssistant(input: TimeManagementAssistantInput): Promise<TimeManagementAssistantOutput> {
  return timeManagementAssistantFlow(input);
}

const prompt = ai.definePrompt({
  name: 'timeManagementAssistantPrompt',
  input: {schema: TimeManagementAssistantInputSchema},
  output: {schema: TimeManagementAssistantOutputSchema},
  prompt: `You are a time management assistant for meetings.  When an agenda item is running over the allocated time, you will provide suggestions on how to get back on track.  If it is not over time, you will indicate that it is not over time and provide null suggestions.

Agenda Item: {{{agendaItem}}}
Time Allocated: {{{timeAllocated}}} minutes
Time Elapsed: {{{timeElapsed}}} minutes

Respond with suggestions only when the time elapsed exceeds the allocated time.
`,
});

const timeManagementAssistantFlow = ai.defineFlow(
  {
    name: 'timeManagementAssistantFlow',
    inputSchema: TimeManagementAssistantInputSchema,
    outputSchema: TimeManagementAssistantOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    const isOverTime = input.timeElapsed > input.timeAllocated;

    return {
      isOverTime: isOverTime,
      suggestions: isOverTime ? output?.suggestions ?? [] : [],
    };
  }
);
