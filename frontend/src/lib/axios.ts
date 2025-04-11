import axios from 'axios'
import { getSession } from 'next-auth/react'

export const api = axios.create({
  baseURL: 'http://127.0.0.1:8000/api',
  headers: {
    'Content-Type': 'application/json',
  },
})

api.interceptors.request.use(async (config) => {
  const session = await getSession()

  if (session?.id_token) {
    config.headers.Authorization = `Bearer ${session.id_token}`
  }

  return config
})
