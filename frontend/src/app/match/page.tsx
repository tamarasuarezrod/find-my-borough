'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';

const questions = [
  {
    id: 'budget_weight',
    title: '💰 Rent prices',
    description: 'How sensitive are you to rent prices?',
    options: [
      { label: "I'm on a tight budget – affordability is key", value: 1 },
      { label: "I care about price, but I can stretch a bit", value: 0.5 },
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
      { label: "Not a big concern for me", value: 0 },
    ],
  },
  {
    id: 'centrality_weight',
    title: '📍 Location',
    description: 'How important is it for you to live close to central London?',
    options: [
      { label: "I want to be in the heart of the city", value: 1 },
      { label: "It’d be nice, but I’m flexible", value: 0.5 },
      { label: "I don’t mind being further out", value: 0 },
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
];

export default function MatchPage() {
  const router = useRouter();
  const [answers, setAnswers] = useState<Record<string, any>>({});
  const [error, setError] = useState<string | null>(null);

  const handleSelect = (questionId: string, value: any) => {
    setAnswers((prev) => ({ ...prev, [questionId]: value }));
    setError(null);
  };

  const handleSubmit = async () => {
    const defaultValues: Record<string, any> = {
      budget_weight: 0,
      safety_weight: 0,
      centrality_weight: 0,
      youth_weight: 0,
      stay_duration: 'unknown',
      is_student: false,
    };

    const payload = {
      ...defaultValues,
      ...answers,
    };

    try {
      const res = await fetch('http://127.0.0.1:8000/api/recommendations/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Requested-With': 'XMLHttpRequest',
        },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error('Failed to fetch recommendations');
      }

      const data = await res.json();
      sessionStorage.setItem('recommendations', JSON.stringify(data));
      router.push('/match/results', { scroll: true });
    } catch (err: any) {
      setError(err.message || 'Something went wrong');
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 py-10">
      <h1 className="text-3xl font-semibold text-center mb-2">Find Your Match</h1>
      <p className="text-gray-400 text-center mb-10">Answer a few questions to find your ideal London borough</p>

      {error && <p className="text-red-400 text-center mb-6">{error}</p>}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {questions.map((q) => (
          <div key={q.id} className="bg-zinc-900 p-6 rounded-xl">
            <h2 className="text-lg font-medium mb-1">{q.title}</h2>
            <p className="text-sm text-gray-400 mb-4">{q.description}</p>
            <div className="flex flex-col gap-2">
              {q.options.map((opt) => (
                <button
                  key={opt.label}
                  onClick={() => handleSelect(q.id, opt.value)}
                  className={`text-left px-4 py-2 rounded-lg border ${
                    answers[q.id] === opt.value
                      ? 'bg-white text-black border-white'
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

      <div className="flex justify-center mt-10">
        <button
          onClick={handleSubmit}
          className="bg-white text-black px-8 py-3 rounded-full font-medium shadow hover:scale-105 transition"
        >
          Find my match →
        </button>
      </div>
    </div>
  );
}
