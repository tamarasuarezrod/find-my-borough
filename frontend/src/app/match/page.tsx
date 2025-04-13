'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { useRecommendation } from '@/services/get-recommendation'
import { useSaveUserAnswers } from '@/services/save-answers'
import { MatchQuestion, useMatchQuestions } from '@/services/get-questions'
import { useSession } from 'next-auth/react'
import LoginModal from '@/components/login-modal'
import { UserAnswers } from '@/types/borough'
import { showErrorToast } from '@/lib/utils'
import { Loader } from '@/components/loader'
import { useAuth } from '@/context/auth-context'

export default function MatchPage() {
  const { status } = useSession()
  const router = useRouter()
  const [answers, setAnswers] = useState<
    Record<string, number | string | boolean>
  >({})
  const [error, setError] = useState<string | null>(null)

  const { mutateAsync: fetchRecommendation, isPending: isLoadingRec } =
    useRecommendation()
  const { mutateAsync: saveAnswers, isPending: isSavingAnswers } =
    useSaveUserAnswers()
  const { data: questions, isLoading } = useMatchQuestions()
  const { isAuthenticated } = useAuth()

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

    saveAnswers(answers, {
      onSuccess: () => {
        fetchRecommendation(answers as UserAnswers, {
          onSuccess: (data) => {
            sessionStorage.setItem('recommendations', JSON.stringify(data))
            router.push('/match/results', { scroll: true })
          },
          onError: () => {
            showErrorToast('Something went wrong')
          },
        })
      },
    })
  }

  const closeModal = () => {
    router.push('/')
  }

  if (isLoading || !questions) {
    return (
      <div className="mt-8 flex items-center justify-center gap-3 text-gray-400">
        <span>Loading questions..</span>
        <Loader />
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-10">
      {!isAuthenticated && <LoginModal onClose={closeModal} />}

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
          {isLoadingRec || isSavingAnswers ? (
            <Loader />
          ) : (
            <button
              onClick={handleSubmit}
              className="rounded-full bg-white px-8 py-3 font-medium text-black shadow transition hover:scale-105"
            >
              Find my match →
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
