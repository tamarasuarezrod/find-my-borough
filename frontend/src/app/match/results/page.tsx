'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';

type Borough = {
  borough: string;
  score: number;
  norm_rent: number;
  norm_safety: number;
  norm_youth: number;
  norm_centrality: number;
};

export default function ResultsPage() {
  const [data, setData] = useState<Borough[]>([]);

  useEffect(() => {
    const stored = sessionStorage.getItem('recommendations');
    if (stored) {
      setData(JSON.parse(stored));
    }
  }, []);

  return (
    <div className="max-w-6xl mx-auto px-6 py-10">
      <h1 className="text-3xl font-bold mb-6">Your Top Borough Matches</h1>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
        {data.map((b, i) => (
          <Link
            key={i}
            href={`/borough/${b.borough.toLowerCase().replace(/ /g, '-')}`}
            className="bg-zinc-900 rounded-xl p-5 hover:scale-105 transition"
          >
            <h2 className="text-xl font-semibold mb-2 capitalize">{b.borough}</h2>
            <p className="text-gray-400 text-sm mb-1">🏆 Score: {(b.score * 100).toFixed(1)}%</p>
            <p className="text-gray-400 text-sm">💰 Rent: {(b.norm_rent * 100).toFixed(0)}%</p>
            <p className="text-gray-400 text-sm">🛡️ Safety: {(b.norm_safety * 100).toFixed(0)}%</p>
            <p className="text-gray-400 text-sm">👥 Youth: {(b.norm_youth * 100).toFixed(0)}%</p>
            <p className="text-gray-400 text-sm">📍 Centrality: {(b.norm_centrality * 100).toFixed(0)}%</p>
          </Link>
        ))}
      </div>
    </div>
  );
}
