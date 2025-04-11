'use client'

import { SessionProvider } from 'next-auth/react'
import { BoroughsProvider } from '@/context/boroughs-context'
import ToastProvider from '@/components/toast-provider'
import Link from 'next/link'
import Providers from '@/app/providers'
import UserActions from '@/components/user-actions'

export default function ClientProviders({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <SessionProvider>
      <Providers>
        <header className="flex flex-col items-center justify-between px-6 py-4 sm:flex-row">
          <Link href="/" className="mb-4 text-xl font-bold sm:mb-0">
            FindMyBorough
          </Link>
          <div className="flex items-center gap-6">
            <nav className="flex gap-4">
              <Link href="/match" className="hover:underline">
                Your match
              </Link>
              <Link href="/explore" className="hover:underline">
                Explore
              </Link>
              <Link href="/about" className="hover:underline">
                About
              </Link>
            </nav>
            <UserActions />
          </div>
        </header>

        <main>
          <BoroughsProvider>
            {children}
            <ToastProvider />
          </BoroughsProvider>
        </main>
      </Providers>
    </SessionProvider>
  )
}
