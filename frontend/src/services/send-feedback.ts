import { api } from '@/lib/axios'
import { useMutation } from '@tanstack/react-query'

type FeedbackPayload = {
  borough: string
  feedback: boolean
}

export const sendFeedback = async (data: FeedbackPayload) => {
  await api.post('/match/feedback/', data)
}

export const useSendFeedback = () => {
  return useMutation({
    mutationFn: sendFeedback,
  })
}
