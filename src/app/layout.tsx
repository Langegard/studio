import type { Metadata } from 'next';
import { GeistSans } from 'geist/font/sans';
import { GeistMono } from 'geist/font/mono';
import './globals.css';
import { Toaster } from "@/components/ui/toaster";

export const metadata: Metadata = {
  title: 'TimeWise Meeting',
  description: 'Track your meeting agenda and keep time effectively.',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${GeistSans.variable} ${GeistMono.variable} h-full`}>
      <body className="antialiased flex flex-col min-h-screen">
        <main className="flex-grow flex flex-col">{children}</main>
        <Toaster />
      </body>
    </html>
  );
}
