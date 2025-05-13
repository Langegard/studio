
"use client";

import type { AgendaItem } from "@/lib/types";
import { timeManagementAssistant, type TimeManagementAssistantOutput } from "@/ai/flows/time-management-assistant";
import { useEffect, useState, useMemo, useCallback } from "react";
import { getAgendaItemColor } from "@/components/agenda-colors";
import { Button } from "@/components/ui/button";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { AlertTriangle, Info, Timer, ChevronLeft } from "lucide-react";
import { useRouter } from "next/navigation";
import { Progress } from "@/components/ui/progress";

interface MeetingTimelineProps {
  agendaItems: AgendaItem[];
}

const TIME_RESOLUTION_MS = 100; // Update timer every 100ms for smoother animation

// Helper to determine text color based on background luminance
function getLuminance(colorString: string): number {
  // Attempt to parse HSL values like "H S% L%" (common from getComputedStyle for HSL vars)
  // or "H, S%, L%". This regex handles optional commas and variable spacing.
  const directHslMatch = colorString.match(/^\s*(\d{1,3})\s*[, ]?\s*(\d{1,3})%\s*[, ]?\s*(\d{1,3})%\s*$/i);
  if (directHslMatch) {
    const l = parseInt(directHslMatch[3], 10) / 100;
    return l; // L value from 0 to 1
  }

  // Attempt to parse standard CSS HSL format: hsl(H, S%, L%) or hsl(H S% L%)
  const cssHslMatch = colorString.match(/hsl\(?\s*(\d{1,3})\s*[, ]?\s*(\d{1,3})%\s*[, ]?\s*(\d{1,3})%\s*\)?/i);
  if (cssHslMatch) {
    const l = parseInt(cssHslMatch[3], 10) / 100;
    return l; // L value from 0 to 1
  }

  // If it's a CSS variable string like 'hsl(var(--some-color))', resolve it
  const varMatch = colorString.match(/hsl\(var\((--[^)]+)\)\)/i);
  if (varMatch && typeof window !== 'undefined') {
    const varName = varMatch[1];
    // Ensure documentElement is available (client-side check)
    if (document?.documentElement) {
      const computedColorValue = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
      // Recursively call with the resolved value (which should be "H S% L%" or similar)
      // Guard against infinite loops if resolution fails or var is not HSL
      if (computedColorValue && computedColorValue !== colorString) {
        return getLuminance(computedColorValue);
      }
    }
  }
  // Fallback for unparsable colors or if not client-side for var resolution
  // console.warn(`Could not parse color string for luminance: "${colorString}"`);
  return 0.5; // Default to mid luminance
}

const LIGHT_TEXT_COLOR = 'hsl(0, 0%, 95%)'; // Very light gray, almost white
const DARK_TEXT_COLOR = 'hsl(0, 0%, 5%)';   // Very dark gray, almost black

export default function MeetingTimeline({ agendaItems }: MeetingTimelineProps) {
  const router = useRouter();
  const [totalElapsedTimeMs, setTotalElapsedTimeMs] = useState(0);
  const [currentAgendaItemIndex, setCurrentAgendaItemIndex] = useState(0);
  const [aiAlert, setAiAlert] = useState<TimeManagementAssistantOutput | null>(null);
  const [showAiAlertDialog, setShowAiAlertDialog] = useState(false);
  const [lastAiCheckTime, setLastAiCheckTime] = useState<Record<number, number>>({}); // Tracks last AI check per item elapsed time
  const [resolvedTextColors, setResolvedTextColors] = useState<string[]>([]);


  const totalMeetingDurationMinutes = useMemo(
    () => agendaItems.reduce((sum, item) => sum + item.time, 0),
    [agendaItems]
  );
  const totalMeetingDurationMs = totalMeetingDurationMinutes * 60 * 1000;

  useEffect(() => {
    if (typeof window !== 'undefined' && document?.documentElement) {
      const colors = agendaItems.map((_, index) => {
        const itemBgColor = getAgendaItemColor(index);
        const luminance = getLuminance(itemBgColor);
        // If luminance is > 0.5 (lighter background), use dark text.
        // Otherwise (darker background), use light text.
        return luminance > 0.5 ? DARK_TEXT_COLOR : LIGHT_TEXT_COLOR;
      });
      setResolvedTextColors(colors);
    }
  }, [agendaItems]);


  useEffect(() => {
    const timer = setInterval(() => {
      setTotalElapsedTimeMs((prev) => {
        const newTime = prev + TIME_RESOLUTION_MS;
        return newTime > totalMeetingDurationMs ? totalMeetingDurationMs : newTime;
      });
    }, TIME_RESOLUTION_MS);

    return () => clearInterval(timer);
  }, [totalMeetingDurationMs]);

  const { currentItemElapsedTimeMs, cumulativeTimeUpToCurrentItemMs } = useMemo(() => {
    let cumulativeTime = 0;
    for (let i = 0; i < agendaItems.length; i++) {
      const itemDurationMs = agendaItems[i].time * 60 * 1000;
      if (totalElapsedTimeMs < cumulativeTime + itemDurationMs || i === agendaItems.length - 1) {
        setCurrentAgendaItemIndex(i);
        return { 
          currentItemElapsedTimeMs: Math.max(0, totalElapsedTimeMs - cumulativeTime),
          cumulativeTimeUpToCurrentItemMs: cumulativeTime
        };
      }
      cumulativeTime += itemDurationMs;
    }
    // Should not be reached if agendaItems is not empty
    setCurrentAgendaItemIndex(agendaItems.length > 0 ? agendaItems.length -1 : 0);
    return { 
      currentItemElapsedTimeMs: agendaItems.length > 0 ? Math.max(0, totalElapsedTimeMs - cumulativeTime + (agendaItems[agendaItems.length-1].time * 60 * 1000)) : 0,
      cumulativeTimeUpToCurrentItemMs: cumulativeTime - (agendaItems.length > 0 ? (agendaItems[agendaItems.length-1].time * 60 * 1000) : 0)
    };
  }, [totalElapsedTimeMs, agendaItems]);


  const checkAiAssistant = useCallback(async () => {
    if (currentAgendaItemIndex < 0 || currentAgendaItemIndex >= agendaItems.length) return;

    const currentItem = agendaItems[currentAgendaItemIndex];
    const allocatedTimeMinutes = currentItem.time;
    const elapsedMinutesInCurrentItem = Math.floor(currentItemElapsedTimeMs / (60 * 1000));

    // Check AI only once per minute for the current item to avoid spamming
    if (lastAiCheckTime[currentAgendaItemIndex] === elapsedMinutesInCurrentItem) {
      return;
    }
    
    // Check slightly before or when time is up
    if (elapsedMinutesInCurrentItem >= allocatedTimeMinutes -1 || elapsedMinutesInCurrentItem > allocatedTimeMinutes) {
       try {
        setLastAiCheckTime(prev => ({...prev, [currentAgendaItemIndex]: elapsedMinutesInCurrentItem}));
        const result = await timeManagementAssistant({
          agendaItem: currentItem.headline,
          timeAllocated: allocatedTimeMinutes,
          timeElapsed: elapsedMinutesInCurrentItem,
        });
        if (result.isOverTime && result.suggestions.length > 0) {
          setAiAlert(result);
          setShowAiAlertDialog(true);
        }
      } catch (error) {
        console.error("Error calling AI assistant:", error);
        // Optionally, show a toast or small error message to the user
      }
    }
  }, [currentAgendaItemIndex, agendaItems, currentItemElapsedTimeMs, lastAiCheckTime]);

  useEffect(() => {
    checkAiAssistant();
  }, [checkAiAssistant]);

  const timerLinePositionPercent = totalMeetingDurationMs > 0 ? (totalElapsedTimeMs / totalMeetingDurationMs) * 100 : 0;

  const formatTime = (ms: number) => {
    const totalSeconds = Math.floor(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  };
  
  const currentItem = agendaItems[currentAgendaItemIndex];
  const currentItemAllocatedMs = currentItem ? currentItem.time * 60 * 1000 : 0;
  const currentItemProgress = currentItemAllocatedMs > 0 ? (currentItemElapsedTimeMs / currentItemAllocatedMs) * 100 : 0;


  return (
    <div className="flex flex-col h-full bg-background text-foreground p-4 sm:p-6 lg:p-8 relative overflow-hidden">
      <div className="mb-6 flex justify-between items-center">
        <Button variant="outline" onClick={() => router.push('/')} className="shadow-md">
          <ChevronLeft className="mr-2 h-5 w-5" /> Back to Agenda
        </Button>
        <div className="text-right">
          <h1 className="text-2xl font-bold text-primary">TimeWise Meeting</h1>
          <p className="text-muted-foreground">
            Total Time: {formatTime(totalElapsedTimeMs)} / {formatTime(totalMeetingDurationMs)}
          </p>
        </div>
      </div>

      {currentItem && (
        <div className="mb-6 p-4 bg-card rounded-lg shadow-lg border border-accent">
          <h2 className="text-xl sm:text-2xl font-semibold text-accent mb-2">Current: {currentItem.headline}</h2>
          <div className="flex justify-between items-center text-sm sm:text-base text-muted-foreground mb-1">
            <span>Time for this item: {formatTime(currentItemElapsedTimeMs)} / {formatTime(currentItemAllocatedMs)}</span>
            <span>{Math.max(0, currentItem.time - Math.ceil(currentItemElapsedTimeMs / (60 * 1000)))} min remaining</span>
          </div>
          <Progress value={currentItemProgress} className="w-full h-3" />
        </div>
      )}
      
      <div className="flex-grow flex flex-col justify-center">
        <div className="relative w-full h-20 sm:h-24 md:h-32 bg-muted rounded-lg shadow-inner overflow-hidden border">
          {/* Timeline items */}
          <div className="flex h-full">
            {agendaItems.map((item, index) => {
              const itemDurationMs = item.time * 60 * 1000;
              const itemWidthPercent = totalMeetingDurationMs > 0 ? (itemDurationMs / totalMeetingDurationMs) * 100 : 0;
              
              const itemStartTimeMs = agendaItems.slice(0, index).reduce((sum, curr) => sum + curr.time * 60 * 1000, 0);
              const itemEndTimeMs = itemStartTimeMs + itemDurationMs;
              const isPassed = totalElapsedTimeMs >= itemEndTimeMs;
              const isActive = currentAgendaItemIndex === index;
              const textColor = resolvedTextColors[index] || DARK_TEXT_COLOR; // Default to dark text if not resolved

              return (
                <div
                  key={item.id}
                  className={`h-full flex flex-col items-center justify-center transition-all duration-200 ease-linear relative overflow-hidden p-1 sm:p-2 ${isActive ? 'ring-2 ring-accent ring-inset z-10' : ''}`}
                  style={{
                    width: `${itemWidthPercent}%`,
                    backgroundColor: isPassed ? 'hsl(var(--muted))' : getAgendaItemColor(index),
                    filter: isPassed ? 'grayscale(80%) brightness(0.9)' : 'none',
                  }}
                  title={`${item.headline} (${item.time} min)`}
                >
                  <span 
                    className={`text-xs sm:text-sm font-medium truncate text-center`}
                    style={{
                      color: isPassed ? 'hsl(var(--muted-foreground))' : textColor
                    }}
                  >
                    {item.headline}
                  </span>
                   <span className={`text-[0.6rem] sm:text-xs`}
                     style={{
                      color: isPassed ? 'hsl(var(--muted-foreground))' : textColor,
                      opacity: isPassed? 0.6 : 0.8,
                    }}
                   >
                    {item.time} min
                  </span>
                </div>
              );
            })}
          </div>

          {/* Timer line */}
          <div
            className="absolute top-0 bottom-0 w-1.5 bg-foreground shadow-2xl rounded-full"
            style={{
              left: `${timerLinePositionPercent}%`,
              transition: `left ${TIME_RESOLUTION_MS}ms linear`,
            }}
          />
        </div>
      </div>


      {aiAlert && (
        <AlertDialog open={showAiAlertDialog} onOpenChange={setShowAiAlertDialog}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle className="flex items-center">
                <AlertTriangle className="text-destructive mr-2 h-6 w-6" />
                Time Alert: "{agendaItems[currentAgendaItemIndex]?.headline}"
              </AlertDialogTitle>
              <AlertDialogDescription className="text-base py-2">
                This item is running over its allocated time.
                {aiAlert.suggestions.length > 0 && (
                  <>
                    <br />
                    <strong className="mt-2 block">Suggestions:</strong>
                  </>
                )}
                <ul className="list-disc pl-5 space-y-1 mt-1">
                  {aiAlert.suggestions.map((suggestion, i) => (
                    <li key={i}>{suggestion}</li>
                  ))}
                </ul>
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogAction onClick={() => setShowAiAlertDialog(false)}>
                Got it
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      )}
      
      <footer className="mt-auto pt-6 text-center text-sm text-muted-foreground">
        <p>&copy; {new Date().getFullYear()} TimeWise Meeting. Stay focused and on track.</p>
      </footer>
    </div>
  );
}

