import Head from 'next/head';
import Image from 'next/image';
import Link from 'next/link';

const name = 'Wei Jang';
export const siteTitle = 'Wei Jiang blog';

export default function Layout({ children, home }: { children: React.ReactNode, home: boolean }) {
  return (
    <div className="container pt-20 mx-auto sm:pt-10 text-lg max-w-2xl">
      <Head>
        <link rel="icon" href="/favicon.ico" />
        <meta
          name="description"
          content="Learn how to build a personal website using Next.js"
        />
        <meta
          property="og:image"
          content={`https://og-image.vercel.app/${encodeURI(
            siteTitle,
          )}.png?theme=light&md=0&fontSize=75px&images=https%3A%2F%2Fassets.zeit.co%2Fimage%2Fupload%2Ffront%2Fassets%2Fdesign%2Fnextjs-black-logo.svg`}
        />
        <meta name="og:title" content={siteTitle} />
        <meta name="twitter:card" content="summary_large_image" />
      </Head>
      <header className="flex flex-col items-center">
        {home ? (
          <>
            <h1 className="text-4xl sm:text-3xl text-center">{name}</h1> {/* Adjusted for smaller screens */}
          </>
        ) : (
          <>
            <h2 className="text-2xl sm:text-xl text-center"> {/* Adjusted for smaller screens */}
              <Link href="/" className="text-inherit">
                {siteTitle}
              </Link>
            </h2>
          </>
        )}
      </header>
      <main>{children}</main>
      {!home && (
        <div className="mt-4">
          <Link href="/" className="text-blue-500">← Back to home</Link>
        </div>
      )}
    </div>
  );
}
