import { api } from '@/lib/axios'
import { useMutation } from '@tanstack/react-query'

export type MatchAnswers = {
  [questionId: string]: string | number | boolean
}

export const saveUserAnswers = async (answers: MatchAnswers): Promise<void> => {
  await api.post('/match/answers/', { answers })
}

export const useSaveUserAnswers = () =>
  useMutation({
    mutationFn: saveUserAnswers,
  })
