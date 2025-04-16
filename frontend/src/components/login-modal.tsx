'use client'

import { useAuth } from '@/context/auth-context'
import { X } from 'lucide-react'
import Image from 'next/image'
import { useRouter } from 'next/navigation'

interface LoginModalProps {
  onClose: () => void
}

const LoginModal: React.FC<LoginModalProps> = ({ onClose }) => {
  const router = useRouter()
  const { loginWithGoogle, loginWithFacebook } = useAuth()

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

        <div className="space-y-4">
          <button
            onClick={loginWithGoogle}
            className="flex w-full items-center justify-center gap-3 rounded-full bg-white px-6 py-3 text-sm font-semibold text-zinc-800 shadow-sm transition hover:bg-gray-200"
          >
            <Image
              src="/google-logo.svg"
              alt="Google logo"
              width={18}
              height={18}
            />
            Sign in with Google
          </button>
          {process.env.NODE_ENV === 'development' && (
            <button
              onClick={() => {
                console.log(window.location.href)
                loginWithFacebook()
              }}
              className="flex w-full items-center justify-center gap-3 rounded-full bg-[#155ec9] px-6 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-[#1a6dd8]"
            >
              <Image
                src="/facebook-logo.svg"
                alt="Facebook logo"
                width={20}
                height={20}
              />
              Sign in with Facebook
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

export default LoginModal
