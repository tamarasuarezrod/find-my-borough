import ClientProviders from '@/context/client-providers'
import { Sora } from 'next/font/google'
import '../styles/globals.css'

const sora = Sora({
  subsets: ['latin'],
  weight: ['300', '400', '500', '600', '700'],
  variable: '--font-sora',
})

export const metadata = {
  title: 'FindMyBorough',
  description: 'Find the perfect borough to live in London',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={`${sora.className} bg-black font-sans text-white`}>
        <ClientProviders>{children}</ClientProviders>
      </body>
    </html>
  )
}
