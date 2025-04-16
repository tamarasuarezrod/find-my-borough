'use client'

import { useAuth } from '@/context/auth-context'
import { User } from 'lucide-react'
import { useSession } from 'next-auth/react'
import Image from 'next/image'
import { useEffect, useRef, useState } from 'react'

export default function UserActions() {
  const { data: session, status } = useSession()
  const [menuOpen, setMenuOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)
  const { isAuthenticated, loginWithGoogle, logout, isLoadingLogin } = useAuth()

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setMenuOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  if (status === 'loading') return null

  if (!isAuthenticated && !isLoadingLogin) {
    return (
      <button
        onClick={loginWithGoogle}
        className="whitespace-nowrap rounded-full border border-white px-5 py-2 text-sm transition hover:bg-white hover:text-black"
      >
        Log in
      </button>
    )
  }

  return (
    <div className="relative" ref={menuRef}>
      <button
        onClick={() => setMenuOpen((prev) => !prev)}
        className="ml-4 flex h-8 w-8 items-center justify-center overflow-hidden rounded-full border-2 border-white transition hover:scale-105"
      >
        {session?.user?.image ? (
          <Image
            src={session.user.image}
            alt="Profile"
            width={32}
            height={32}
            className="h-8 w-8 rounded-full object-cover"
          />
        ) : (
          <User className="h-4 w-4 text-white" />
        )}
      </button>

      {menuOpen && (
        <div className="absolute right-0 z-10 mt-2 w-32 rounded-md bg-zinc-800 p-2 shadow-lg">
          <button
            onClick={logout}
            className="w-full rounded px-3 py-2 text-left text-sm hover:bg-zinc-700"
          >
            Logout
          </button>
        </div>
      )}
    </div>
  )
}
