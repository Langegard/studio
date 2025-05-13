"use client";

import MeetingTimeline from "@/components/meeting-timeline";
import type { AgendaItem } from "@/lib/types";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { AlertCircle, Loader2 } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import Link from "next/link";

function MeetingPageContent() {
  const searchParams = useSearchParams();
  const agendaJson = searchParams.get("agenda");

  if (!agendaJson) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center p-8">
        <Alert variant="destructive" className="max-w-md">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error: No Agenda Data</AlertTitle>
          <AlertDescription>
            Meeting agenda not found. Please go back and create an agenda first.
          </AlertDescription>
        </Alert>
        <Button asChild className="mt-6">
          <Link href="/">Create Agenda</Link>
        </Button>
      </div>
    );
  }

  let agendaItems: AgendaItem[];
  try {
    agendaItems = JSON.parse(agendaJson);
    if (!Array.isArray(agendaItems) || agendaItems.length === 0 || !agendaItems.every(item => item.headline && typeof item.time === 'number')) {
      throw new Error("Invalid agenda format");
    }
  } catch (error) {
    return (
       <div className="flex flex-col items-center justify-center h-full text-center p-8">
        <Alert variant="destructive" className="max-w-md">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error: Invalid Agenda Data</AlertTitle>
          <AlertDescription>
            The provided agenda data is corrupted or invalid. Please go back and create a new agenda.
          </AlertDescription>
        </Alert>
         <Button asChild className="mt-6">
          <Link href="/">Create Agenda</Link>
        </Button>
      </div>
    );
  }

  return <MeetingTimeline agendaItems={agendaItems} />;
}


export default function MeetingPage() {
  return (
    <Suspense fallback={<LoadingState />}>
      <MeetingPageContent />
    </Suspense>
  );
}

function LoadingState() {
  return (
    <div className="flex flex-col items-center justify-center h-full p-8">
      <Loader2 className="h-12 w-12 animate-spin text-primary mb-4" />
      <p className="text-xl text-muted-foreground">Loading Meeting Timeline...</p>
    </div>
  )
}
