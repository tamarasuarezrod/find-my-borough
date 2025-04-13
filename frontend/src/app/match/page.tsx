'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { useRecommendation } from '@/services/get-recommendation'
import { useSaveUserAnswers } from '@/services/save-answers'
import { MatchQuestion, useMatchQuestions } from '@/services/get-questions'
import { useSession } from 'next-auth/react'
import LoginModal from '@/components/login-modal'
import { UserAnswers } from '@/types/borough'
import { showErrorToast } from '@/lib/utils'

export default function MatchPage() {
  const { status } = useSession()
  const router = useRouter()
  const [answers, setAnswers] = useState<
    Record<string, number | string | boolean>
  >({})
  const [error, setError] = useState<string | null>(null)
  const [showLoginModal, setShowLoginModal] = useState(false)

  const { mutateAsync: fetchRecommendation } = useRecommendation()
  const saveAnswers = useSaveUserAnswers()
  const { data: questions, isLoading } = useMatchQuestions()

  useEffect(() => {
    if (status === 'unauthenticated') {
      setShowLoginModal(true)
    }
  }, [status])

  const handleSelect = (
    questionId: string,
    value: number | string | boolean,
  ) => {
    setAnswers((prev) => ({ ...prev, [questionId]: value }))
    setError(null)
  }

  const handleSubmit = async () => {
    const answeredCount = Object.keys(answers).length

    if (answeredCount < 4) {
      showErrorToast('Please answer at least 4 questions before continuing')
      return
    }

    try {
      if (status === 'authenticated') {
        await saveAnswers.mutateAsync(answers)
      }

      const data = await fetchRecommendation(answers as UserAnswers)
      sessionStorage.setItem('recommendations', JSON.stringify(data))
      router.push('/match/results', { scroll: true })
    } catch {
      showErrorToast('Something went wrong')
    }
  }

  const closeModal = () => {
    setShowLoginModal(false)
    router.push('/')
  }

  if (isLoading || !questions) {
    return <div className="text-center text-gray-400">Loading questions...</div>
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-10">
      {showLoginModal && <LoginModal onClose={closeModal} />}

      <h1 className="mb-2 text-center text-3xl font-semibold">
        Find Your Match
      </h1>
      <p className="mb-10 text-center text-gray-400">
        Answer a few questions to find your ideal London borough
      </p>

      {error && <p className="mb-6 text-center text-red-400">{error}</p>}

      <div>
        <div className="grid grid-cols-1 gap-8 md:grid-cols-2">
          {questions.map((q: MatchQuestion) => (
            <div key={q.id} className="md:rounded-xl md:bg-zinc-900 md:p-6">
              <h2 className="mb-1 text-lg font-medium">{q.title}</h2>
              <p className="mb-4 text-sm text-gray-400">{q.description}</p>
              <div className="flex flex-col gap-2">
                {q.options.map((opt) => (
                  <button
                    key={opt.label}
                    onClick={() => handleSelect(q.id, opt.value)}
                    className={`rounded-lg border px-4 py-2 text-left ${
                      answers[q.id] === opt.value
                        ? 'border-white bg-white text-black'
                        : 'border-zinc-700 hover:bg-zinc-800'
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="mt-10 flex justify-center">
          <button
            onClick={handleSubmit}
            className="rounded-full bg-white px-8 py-3 font-medium text-black shadow transition hover:scale-105"
          >
            Find my match →
          </button>
        </div>
      </div>
    </div>
  )
}
