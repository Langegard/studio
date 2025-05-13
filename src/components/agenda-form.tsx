"use client";

import type { AgendaItem } from "@/lib/types";
import { zodResolver } from "@hookform/resolvers/zod";
import { useRouter } from "next/navigation";
import { useFieldArray, useForm } from "react-hook-form";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle, CardFooter, CardDescription } from "@/components/ui/card";
import { PlusCircle, Trash2, Play } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";

const agendaItemSchema = z.object({
  id: z.string(),
  headline: z.string().min(1, "Headline is required.").max(100, "Headline too long."),
  time: z.coerce.number().min(1, "Time must be at least 1 minute.").max(180, "Time too long."),
});

const agendaFormSchema = z.object({
  agendaItems: z.array(agendaItemSchema).min(1, "At least one agenda item is required."),
});

type AgendaFormValues = z.infer<typeof agendaFormSchema>;

export default function AgendaForm() {
  const router = useRouter();
  const form = useForm<AgendaFormValues>({
    resolver: zodResolver(agendaFormSchema),
    defaultValues: {
      agendaItems: [{ id: crypto.randomUUID(), headline: "", time: 10 }],
    },
  });

  const { fields, append, remove } = useFieldArray({
    control: form.control,
    name: "agendaItems",
  });

  const onSubmit = (data: AgendaFormValues) => {
    const agendaJson = JSON.stringify(data.agendaItems);
    router.push(`/meeting?agenda=${encodeURIComponent(agendaJson)}`);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-full p-4 sm:p-6 lg:p-8">
      <Card className="w-full max-w-2xl shadow-2xl">
        <CardHeader className="text-center">
          <CardTitle className="text-3xl font-bold">TimeWise Meeting</CardTitle>
          <CardDescription className="text-muted-foreground">
            Plan your meeting agenda and allocate time for each topic.
          </CardDescription>
        </CardHeader>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)}>
            <CardContent className="space-y-6">
              <ScrollArea className="h-[calc(100vh-28rem)] min-h-[10rem] pr-3">
                <div className="space-y-4">
                {fields.map((field, index) => (
                  <Card key={field.id} className="p-4 shadow-md bg-secondary/30">
                    <div className="grid grid-cols-1 gap-4 md:grid-cols-[1fr_auto_auto] md:items-end">
                      <FormField
                        control={form.control}
                        name={`agendaItems.${index}.headline`}
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Headline {index + 1}</FormLabel>
                            <FormControl>
                              <Input placeholder="e.g., Patient History" {...field} />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />
                      <FormField
                        control={form.control}
                        name={`agendaItems.${index}.time`}
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Time (min)</FormLabel>
                            <FormControl>
                              <Input type="number" placeholder="e.g., 15" {...field} />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />
                      {fields.length > 1 && (
                        <Button
                          type="button"
                          variant="destructive"
                          size="icon"
                          onClick={() => remove(index)}
                          aria-label="Remove agenda item"
                          className="self-end"
                        >
                          <Trash2 />
                        </Button>
                      )}
                    </div>
                  </Card>
                ))}
                </div>
              </ScrollArea>
              <Button
                type="button"
                variant="outline"
                onClick={() => append({ id: crypto.randomUUID(), headline: "", time: 10 })}
                className="w-full"
              >
                <PlusCircle className="mr-2" /> Add Agenda Item
              </Button>
            </CardContent>
            <CardFooter>
              <Button type="submit" className="w-full text-lg py-6" disabled={!form.formState.isValid || form.formState.isSubmitting}>
                <Play className="mr-2" /> Start Meeting
              </Button>
            </CardFooter>
          </form>
        </Form>
      </Card>
    </div>
  );
}
