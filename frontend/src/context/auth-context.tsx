'use client'

import { showErrorToast } from '@/lib/utils'
import { useLoginToBackend } from '@/services/authenticate'
import { signIn, signOut, useSession } from 'next-auth/react'
import { createContext, useContext, useEffect, useRef, useState } from 'react'

type AuthContextType = {
  isAuthenticated: boolean
  loginWithGoogle: () => void
  loginWithFacebook: () => void
  logout: () => void
  isLoadingLogin: boolean
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
  const syncWithBackend = useRef(false)
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const { mutateAsync: loginToBackend, isPending: isLoadingLogin } =
    useLoginToBackend()
  const { data: session } = useSession()

  useEffect(() => {
    const accessToken = localStorage.getItem('access_token')
    const refreshToken = localStorage.getItem('refresh_token')

    if (accessToken && refreshToken) {
      setIsAuthenticated(true)
    } else {
      setIsAuthenticated(false)
    }
  }, [])

  useEffect(() => {
    const sync = async () => {
      const provider = localStorage.getItem('auth_provider') as
        | 'google'
        | 'facebook'
        | null

      if (
        session?.id_token &&
        !localStorage.getItem('access_token') &&
        !syncWithBackend.current &&
        provider
      ) {
        syncWithBackend.current = true
        loginToBackend(
          { provider, token: session?.id_token },
          {
            onSuccess: () => {
              setIsAuthenticated(true)
            },
            onError: () => {
              showErrorToast('There was an error logging in. Please try again.')
            },
          },
        )
      }
    }

    sync()
  }, [loginToBackend, session])

  const loginWithProvider = async (provider: 'google' | 'facebook') => {
    syncWithBackend.current = false
    localStorage.setItem('auth_provider', provider)
    await signIn(provider, { prompt: 'select_account' })
  }

  const logout = async () => {
    await signOut({ redirect: false })
    localStorage.removeItem('access_token')
    localStorage.removeItem('refresh_token')
    setIsAuthenticated(false)
  }

  return (
    <AuthContext.Provider
      value={{
        isAuthenticated,
        loginWithGoogle: () => loginWithProvider('google'),
        loginWithFacebook: () => loginWithProvider('facebook'),
        logout,
        isLoadingLogin,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) throw new Error('useAuth must be used within an AuthProvider')
  return context
}
