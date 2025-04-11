'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { useRecommendation } from '@/services/get-recommendation'
import LoginModal from '@/components/login-modal'
import { useSession } from 'next-auth/react'
import toast from 'react-hot-toast'

const questions = [
  {
    id: 'budget_weight',
    title: '💰 Rent prices',
    description: 'How sensitive are you to rent prices?',
    options: [
      { label: "I'm on a tight budget – affordability is key", value: 1 },
      { label: 'I care about price, but I can stretch a bit', value: 0.5 },
      { label: "I'm willing to pay more for a better area", value: 0 },
    ],
  },
  {
    id: 'safety_weight',
    title: '🛡️ Safety',
    description: 'How much does safety influence your choice of area?',
    options: [
      { label: "I won't compromise on safety", value: 1 },
      { label: "I'd like a safe area, but I'm flexible", value: 0.5 },
      { label: 'Not a big concern for me', value: 0 },
    ],
  },
  {
    id: 'centrality_weight',
    title: '📍 Location',
    description: 'How important is it for you to live close to central London?',
    options: [
      { label: 'I want to be in the heart of the city', value: 1 },
      { label: 'It’d be nice, but I’m flexible', value: 0.5 },
      { label: 'I don’t mind being further out', value: 0 },
    ],
  },
  {
    id: 'youth_weight',
    title: '👥 Youth community',
    description: 'What kind of neighbourhood vibe are you looking for?',
    options: [
      { label: 'Energetic and youthful', value: 1 },
      { label: 'Calm and family-oriented', value: 0.5 },
      { label: 'I don’t mind either way', value: 0 },
    ],
  },
  {
    id: 'stay_duration',
    title: '📅 Duration',
    description: 'How long do you plan to stay?',
    options: [
      { label: 'Less than a year', value: 'short_term' },
      { label: '1–2 years', value: 'mid_term' },
      { label: 'Longer than 2 years', value: 'long_term' },
      { label: 'Not sure yet / Open to anything', value: 'unknown' },
    ],
  },
  {
    id: 'is_student',
    title: '📌 Current situation',
    description: 'What best describes your current situation?',
    options: [
      { label: "I'm a student", value: 'student' },
      { label: "I'm a young professional", value: 'young-professional' },
      { label: 'I’m relocating with family', value: 'family' },
      { label: 'Other', value: 'other' },
    ],
  },
]

export default function MatchPage() {
  const { status } = useSession()
  const router = useRouter()
  const [answers, setAnswers] = useState<
    Record<string, number | boolean | string>
  >({})
  const [error, setError] = useState<string | null>(null)
  const [showLoginModal, setShowLoginModal] = useState(false)

  const { mutateAsync: fetchRecommendation } = useRecommendation()

  useEffect(() => {
    if (status === 'unauthenticated') {
      setShowLoginModal(true)
    }
  }, [status])

  const handleSelect = (
    questionId: string,
    value: number | boolean | string,
  ) => {
    setAnswers((prev) => ({ ...prev, [questionId]: value }))
    setError(null)
  }

  const handleSubmit = async () => {
    const answeredCount = Object.keys(answers).length

    if (answeredCount < 4) {
      toast.error('Please answer at least 4 questions before continuing', {
        duration: 4000,
      })
      return
    }

    const defaultValues = {
      budget_weight: 0,
      safety_weight: 0,
      centrality_weight: 0,
      youth_weight: 0,
      stay_duration: 'unknown',
      is_student: false,
    }

    const payload = {
      ...defaultValues,
      ...answers,
    }

    try {
      const data = await fetchRecommendation(payload)
      sessionStorage.setItem('recommendations', JSON.stringify(data))
      router.push('/match/results', { scroll: true })
    } catch (err) {
      setError(err.message || 'Something went wrong')
    }
  }

  const closeModal = () => {
    setShowLoginModal(false)
    router.push('/')
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
          {questions.map((q) => (
            <div key={q.id} className="rounded-xl bg-zinc-900 p-6">
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
