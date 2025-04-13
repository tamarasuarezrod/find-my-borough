import axios from 'axios'
import { signOut } from 'next-auth/react'

const refreshAccessToken = async (): Promise<string | null> => {
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
  } catch {
    localStorage.removeItem('access_token')
    localStorage.removeItem('refresh_token')
    signOut({ redirect: false })
    return null
  }
}

export const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: false,
})

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config

    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true
      const newAccessToken = await refreshAccessToken()

      if (newAccessToken) {
        originalRequest.headers.Authorization = `Bearer ${newAccessToken}`
        return api(originalRequest)
      }
    }

    return Promise.reject(error)
  },
)
