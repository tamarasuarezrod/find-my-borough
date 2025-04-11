import axios from 'axios'
import { signIn, signOut } from 'next-auth/react'

export const refreshAccessToken = async (): Promise<string | null> => {
  const refresh = localStorage.getItem('refresh_token')
  if (!refresh) return null

  try {
    const res = await axios.post(
      `${process.env.NEXT_PUBLIC_API_URL}/auth/token/refresh/`,
      {
        refresh,
      },
    )

    const newAccess = res.data.access
    localStorage.setItem('access_token', newAccess)
    return newAccess
  } catch (err) {
    console.error('Failed to refresh token', err)
    localStorage.removeItem('access_token')
    localStorage.removeItem('refresh_token')
    return null
  }
}

export const loginWithGoogle = async () => {
  await signIn('google', { prompt: 'select_account' })
}

export const logoutUser = async () => {
  await signOut({ redirect: false })
  localStorage.removeItem('access_token')
  localStorage.removeItem('refresh_token')
}
