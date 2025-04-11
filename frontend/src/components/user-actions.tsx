'use client'

import { useSession } from 'next-auth/react'
import { useState, useEffect, useRef } from 'react'
import Image from 'next/image'
import { useGoogleLogin } from '@/services/post-google-login'
import { loginWithGoogle, logoutUser } from '@/lib/auth'
import { User } from 'lucide-react'

export default function UserActions() {
  const { data: session, status } = useSession()
  const [menuOpen, setMenuOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)
  const { mutateAsync: loginToBackend } = useGoogleLogin()

  useEffect(() => {
    const sync = async () => {
      if (session?.id_token && !localStorage.getItem('access_token')) {
        try {
          await loginToBackend(session.id_token)
        } catch (err) {
          console.error('Sync failed', err)
        }
      }
    }

    sync()
  }, [session?.id_token, loginToBackend])

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

  if (!session) {
    return (
      <button
        onClick={loginWithGoogle}
        className="rounded-full border border-white px-4 py-1 text-sm transition hover:bg-white hover:text-black"
      >
        Log in
      </button>
    )
  }

  return (
    <div className="relative" ref={menuRef}>
      <button
        onClick={() => setMenuOpen((prev) => !prev)}
        className="ml-4 rounded-full border-2 border-white transition hover:scale-105"
      >
        {session.user?.image ? (
          <Image
            src={session.user.image}
            alt="Profile"
            width={32}
            height={32}
            className="h-8 w-8 rounded-full object-cover"
          />
        ) : (
          <div className="flex h-8 w-8 items-center justify-center text-white">
            <User className="h-4 w-4" />
          </div>
        )}
      </button>

      {menuOpen && (
        <div className="absolute right-0 z-10 mt-2 w-32 rounded-md bg-zinc-800 p-2 shadow-lg">
          <button
            onClick={logoutUser}
            className="w-full rounded px-3 py-2 text-left text-sm hover:bg-zinc-700"
          >
            Logout
          </button>
        </div>
      )}
    </div>
  )
}
