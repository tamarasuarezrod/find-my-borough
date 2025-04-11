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
        <header className="flex items-center justify-between px-8 py-5">
          <Link href="/" className="text-l cursor-pointer">
            FindMyBorough
          </Link>
          <nav className="flex items-center space-x-6">
            <Link href="/match" className="hover:underline">
              Your match
            </Link>
            <Link href="/explore" className="hover:underline">
              Explore
            </Link>
            <Link href="/about" className="hover:underline">
              About
            </Link>
            <UserActions />
          </nav>
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
