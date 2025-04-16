import { api } from '@/lib/axios'
import { useMutation } from '@tanstack/react-query'

export type LoginRequest = {
  provider: 'google' | 'facebook'
  token: string
}

export const useLoginToBackend = () => {
  return useMutation({
    mutationFn: async (data: LoginRequest) => {
      const res = await api.post('/auth/google/', data)
      const { access, refresh } = res.data

      localStorage.setItem('access_token', access)
      localStorage.setItem('refresh_token', refresh)

      return res.data
    },
  })
}
