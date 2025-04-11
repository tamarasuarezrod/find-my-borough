'use client'

import { useRouter } from 'next/navigation'
import { X } from 'lucide-react'
import Image from 'next/image'
import { loginWithGoogle } from '@/lib/auth'

interface LoginModalProps {
  onClose: () => void
}

const LoginModal: React.FC<LoginModalProps> = ({ onClose }) => {
  const router = useRouter()

  const handleClose = () => {
    onClose()
    router.push('/')
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="relative w-[90%] max-w-md rounded-2xl bg-zinc-900 p-8 shadow-2xl">
        <button
          onClick={handleClose}
          className="absolute right-4 top-4 text-gray-400 transition hover:text-red-400"
        >
          <X className="h-5 w-5" />
        </button>

        <h2 className="mb-3 text-2xl font-bold text-white">
          Log in to continue
        </h2>
        <p className="mb-6 text-sm text-gray-400">
          You need to be logged in to find your match.
        </p>

        <button
          onClick={loginWithGoogle}
          className="flex w-full items-center justify-center gap-3 rounded-full bg-white px-6 py-3 text-sm font-semibold text-zinc-800 shadow-sm transition hover:bg-gray-100"
        >
          <Image
            src="/google-logo.svg"
            alt="Google logo"
            width={18}
            height={18}
          />
          Sign in with Google
        </button>
      </div>
    </div>
  )
}

export default LoginModal
