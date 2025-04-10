import { Sora } from 'next/font/google';

const sora = Sora({
  subsets: ['latin'],
  weight: ['300', '400', '500', '600', '700'],
  variable: '--font-sora',
});


import '../styles/globals.css';
import Link from 'next/link';

export const metadata = {
  title: 'FindMyBorough',
  description: 'Find the perfect borough to live in London',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${sora.className} bg-black text-white font-sans`}>
        <header className="flex justify-between items-center px-8 py-5">
        <Link href="/" className="text-l cursor-pointer">
          FindMyBorough
        </Link> 
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
