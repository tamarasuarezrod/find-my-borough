import { api } from '@/lib/axios'
import { useMutation } from '@tanstack/react-query'

export type GoogleLoginResponse = {
  token: string
  email: string
  name: string
}

const postGoogleLogin = async (token: string): Promise<GoogleLoginResponse> => {
  const response = await api.post('/account/google/', {
    token,
  })
  return response.data
}

export const useGoogleLogin = () => {
  return useMutation({
    mutationFn: postGoogleLogin,
  })
}
