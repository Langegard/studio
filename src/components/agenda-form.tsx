
"use client";

import type { AgendaItem, Preset } from "@/lib/types";
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
import { PlusCircle, Trash2, Play, GripVertical, Save } from "lucide-react"; // Removed FileUp, FileDown as they are not used
import { ScrollArea } from "@/components/ui/scroll-area";
import React, { useState, useEffect, useCallback } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogClose,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { useToast } from "@/hooks/use-toast";
import { Label } from "@/components/ui/label";

const agendaItemSchema = z.object({
  id: z.string().uuid("Item ID must be a valid UUID."),
  headline: z.string().min(1, "Headline is required.").max(100, "Headline too long."),
  time: z.coerce.number().min(1, "Time must be at least 1 minute.").max(180, "Time too long."),
});

const agendaFormSchema = z.object({
  agendaItems: z.array(agendaItemSchema).min(1, "At least one agenda item is required."),
});

type AgendaFormValues = z.infer<typeof agendaFormSchema>;

const LOCAL_STORAGE_PRESETS_KEY = 'timeWiseMeetingPresets';

export default function AgendaForm() {
  const router = useRouter();
  const { toast } = useToast();
  const form = useForm<AgendaFormValues>({
    resolver: zodResolver(agendaFormSchema),
    defaultValues: {
      agendaItems: [{ id: crypto.randomUUID(), headline: "", time: 10 }],
    },
  });

  const { fields, append, remove, move } = useFieldArray({
    control: form.control,
    name: "agendaItems",
  });

  const [draggedItemIndex, setDraggedItemIndex] = useState<number | null>(null);
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null);

  const [presets, setPresets] = useState<Preset[]>([]);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [newPresetName, setNewPresetName] = useState("");
  const [selectedPresetToLoad, setSelectedPresetToLoad] = useState<string>("");
  const [clientMounted, setClientMounted] = useState(false);

  useEffect(() => {
    setClientMounted(true);
  }, []);

  useEffect(() => {
    if (clientMounted && typeof window !== 'undefined' && window.localStorage) {
      const storedPresets = localStorage.getItem(LOCAL_STORAGE_PRESETS_KEY);
      if (storedPresets) {
        try {
          const parsedPresets = JSON.parse(storedPresets) as Preset[];
          setPresets(parsedPresets);
          if (parsedPresets.length > 0) {
            setSelectedPresetToLoad(parsedPresets[0].name);
          }
        } catch (e) {
          console.error("Failed to parse presets from localStorage", e);
          toast({ title: "Error", description: "Could not load presets.", variant: "destructive" });
        }
      }
    }
  }, [clientMounted, toast]);

  const savePresetsToStorage = useCallback((updatedPresets: Preset[]) => {
    if (typeof window !== 'undefined' && window.localStorage) {
      localStorage.setItem(LOCAL_STORAGE_PRESETS_KEY, JSON.stringify(updatedPresets));
    }
  }, []);

  const handleSavePreset = () => {
    if (!newPresetName.trim()) {
      toast({ title: "Error", description: "Preset name cannot be empty.", variant: "destructive" });
      return;
    }
    const currentAgenda = form.getValues().agendaItems;
    if (currentAgenda.length === 0) {
      toast({ title: "Error", description: "Cannot save an empty agenda.", variant: "destructive" });
      return;
    }

    const existingPresetIndex = presets.findIndex(p => p.name === newPresetName);
    let updatedPresets;
    if (existingPresetIndex !== -1) {
      // Overwrite existing preset
      updatedPresets = [...presets];
      updatedPresets[existingPresetIndex] = { name: newPresetName, agenda: currentAgenda };
      toast({ title: "Preset Updated", description: `Preset "${newPresetName}" has been updated.` });
    } else {
      // Add new preset
      updatedPresets = [...presets, { name: newPresetName, agenda: currentAgenda }];
      toast({ title: "Preset Saved", description: `Preset "${newPresetName}" has been saved.` });
    }
    
    setPresets(updatedPresets);
    savePresetsToStorage(updatedPresets);
    setSelectedPresetToLoad(newPresetName); // Select the newly saved/updated preset
    setShowSaveDialog(false);
    setNewPresetName("");
  };

  const handleLoadPreset = (presetName: string) => {
    if (!presetName) return;
    const presetToLoad = presets.find(p => p.name === presetName);
    if (presetToLoad) {
      const validatedAgendaItems = presetToLoad.agenda.map(item => ({
        ...item,
        id: item.id || crypto.randomUUID(),
      }));
      form.reset({ agendaItems: validatedAgendaItems });
      toast({ title: "Preset Loaded", description: `Agenda for "${presetName}" has been loaded.` });
    } else {
      toast({ title: "Error", description: "Selected preset not found.", variant: "destructive" });
    }
  };
  
  const handleDeletePreset = (presetName: string) => {
    if (!presetName) {
      toast({ title: "Error", description: "No preset selected to delete.", variant: "destructive" });
      return;
    }
    const updatedPresets = presets.filter(p => p.name !== presetName);
    setPresets(updatedPresets);
    savePresetsToStorage(updatedPresets);
    toast({ title: "Preset Deleted", description: `Preset "${presetName}" has been deleted.` });
    if (selectedPresetToLoad === presetName) {
      setSelectedPresetToLoad(updatedPresets.length > 0 ? updatedPresets[0].name : "");
    }
  };


  const handleDragStart = (index: number) => (event: React.DragEvent<HTMLDivElement>) => {
    setDraggedItemIndex(index);
    event.dataTransfer.effectAllowed = 'move';
  };

  const handleDragEnter = (index: number) => (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (draggedItemIndex !== null && index !== draggedItemIndex) {
      setDragOverIndex(index);
    }
  };
  
  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault(); 
    if (draggedItemIndex !== null) {
      event.dataTransfer.dropEffect = 'move';
    }
  };

  const handleDragLeave = () => {
     setDragOverIndex(null);
  };

  const handleDrop = (index: number) => (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (draggedItemIndex !== null && draggedItemIndex !== index) {
      move(draggedItemIndex, index);
    }
    setDraggedItemIndex(null);
    setDragOverIndex(null);
  };

  const handleDragEnd = () => {
    setDraggedItemIndex(null);
    setDragOverIndex(null);
  };

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
            Plan your meeting agenda and allocate time for each topic. Drag items to reorder.
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-4 border-b pb-6">
          <h3 className="text-lg font-medium">Agenda Presets</h3>
          {!clientMounted ? (
            <p className="text-sm text-muted-foreground">Loading preset options...</p>
          ) : (typeof window !== 'undefined' && window.localStorage) ? (
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-[1fr_auto_auto] items-end">
              <div className="flex flex-col">
                <Label htmlFor="load-preset-select" className="mb-1.5 text-sm font-medium">Load Preset</Label>
                 <Select 
                    onValueChange={(value) => { setSelectedPresetToLoad(value); handleLoadPreset(value); }} 
                    value={selectedPresetToLoad}
                  >
                  <SelectTrigger id="load-preset-select" disabled={presets.length === 0}>
                    <SelectValue placeholder="Select a preset to load" />
                  </SelectTrigger>
                  <SelectContent>
                    {presets.map(preset => (
                      <SelectItem key={preset.name} value={preset.name}>{preset.name}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button variant="outline" className="w-full sm:w-auto" disabled={!selectedPresetToLoad || presets.length === 0}>
                    <Trash2 className="mr-2" /> Delete
                  </Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Are you sure?</AlertDialogTitle>
                    <AlertDialogDescription>
                      This will permanently delete the preset "{selectedPresetToLoad}". This action cannot be undone.
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction onClick={() => handleDeletePreset(selectedPresetToLoad)}>Delete</AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
               <Dialog open={showSaveDialog} onOpenChange={setShowSaveDialog}>
                <DialogTrigger asChild>
                  <Button className="w-full sm:w-auto" onClick={() => setShowSaveDialog(true)}>
                    <Save className="mr-2" /> Save Current
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Save Agenda Preset</DialogTitle>
                    <DialogDescription>
                      Enter a name for your current agenda configuration.
                    </DialogDescription>
                  </DialogHeader>
                  <Input 
                    placeholder="Preset name (e.g., Weekly Sync)" 
                    value={newPresetName} 
                    onChange={(e) => setNewPresetName(e.target.value)} 
                    className="my-4"
                  />
                  <DialogFooter>
                    <DialogClose asChild><Button variant="outline">Cancel</Button></DialogClose>
                    <Button onClick={handleSavePreset}>Save Preset</Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">Agenda presets are not available (localStorage is disabled or not accessible).</p>
          )}
        </CardContent>

        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)}>
            <CardContent className="space-y-6 pt-6">
               <ScrollArea className="h-[calc(100vh-38rem)] min-h-[10rem] pr-3">
                <div className="space-y-4">
                {fields.map((field, index) => (
                  <Card 
                    key={field.id} 
                    className={`p-4 shadow-md bg-secondary/30 relative transition-all duration-150 ease-in-out ${draggedItemIndex === index ? 'opacity-50 scale-95 shadow-xl' : ''} ${dragOverIndex === index ? 'ring-2 ring-primary' : ''}`}
                    draggable
                    onDragStart={handleDragStart(index)}
                    onDragEnter={handleDragEnter(index)}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop(index)}
                    onDragEnd={handleDragEnd}
                  >
                    <div className="grid grid-cols-[auto_1fr_auto_auto] gap-x-3 gap-y-4 md:grid-cols-[auto_1fr_auto_auto] md:items-end items-start">
                       <div className="flex items-center justify-center h-full cursor-grab text-muted-foreground hover:text-foreground pt-6 md:pt-0">
                        <GripVertical />
                      </div>
                      <FormField
                        control={form.control}
                        name={`agendaItems.${index}.headline`}
                        render={({ field: controllerField }) => (
                          <FormItem className="w-full">
                            <FormLabel>Headline {index + 1}</FormLabel>
                            <FormControl>
                              <Input placeholder="e.g., Patient History" {...controllerField} />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />
                      <FormField
                        control={form.control}
                        name={`agendaItems.${index}.time`}
                        render={({ field: controllerField }) => (
                          <FormItem>
                            <FormLabel>Time (min)</FormLabel>
                            <FormControl>
                              <Input type="number" placeholder="e.g., 15" {...controllerField} />
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
