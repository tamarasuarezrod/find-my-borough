import '../styles/globals.css';
import Link from 'next/link';

export const metadata = {
  title: 'FindMyBorough',
  description: 'Find the perfect borough to live in London',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-black text-white font-sans">
        <header className="flex justify-between items-center px-8 py-6">
          <h1 className="text-xl font-bold">FindMyBorough</h1>
          <nav className="space-x-6">
            <Link href="/match" className="hover:underline">Your match</Link>
            <Link href="/explore" className="hover:underline">Explore</Link>
            <Link href="/about" className="hover:underline">About</Link>
          </nav>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
