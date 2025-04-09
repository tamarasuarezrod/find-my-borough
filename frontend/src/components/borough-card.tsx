'use client';

import { toTitleCase } from '@/lib/utls';
import Image from 'next/image';
import Link from 'next/link';

type Props = {
  slug: string;
  name: string;
  image: string;
  price?: string;
  safety?: string;
  age?: string;
};

export default function BoroughCard({ slug, name, image, price, safety, age }: Props) {
  return (
    <Link
      href={`/borough/${slug}`}
      className="rounded-2xl overflow-hidden bg-zinc-900 hover:scale-105 transition shadow-lg"
    >
      <div className="relative w-full h-48">
        <Image
          src={image ?? '/images/london-background.png'}
          alt={name}
          fill
          className="object-cover"
        />
      </div>
      <div className="p-4 text-left">
        <h3 className="text-xl font-semibold text-white mb-1">{toTitleCase(name)}</h3>
        {(price || safety || age) && (
          <div className="text-sm text-gray-400 flex gap-4">
            {price && <span>{price}</span>}
            {safety && <span>{safety}</span>}
            {age && <span>{age}</span>}
          </div>
        )}
      </div>
    </Link>
  );
}
