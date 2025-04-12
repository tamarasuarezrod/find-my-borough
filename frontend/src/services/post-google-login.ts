import { api } from '@/lib/axios'
import { useMutation } from '@tanstack/react-query'

export type GoogleLoginResponse = {
  access: string
  refresh: string
  email: string
  name: string
}

const postGoogleLogin = async (
  id_token: string,
): Promise<GoogleLoginResponse> => {
  const res = await api.post('/auth/google/', { token: id_token })

  const { access, refresh } = res.data

  localStorage.setItem('access_token', access)
  localStorage.setItem('refresh_token', refresh)

  return res.data
}

export const useGoogleLogin = () => {
  return useMutation({
    mutationFn: postGoogleLogin,
  })
}
