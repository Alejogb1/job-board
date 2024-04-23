/** @type {import('tailwindcss').Config} */
const withMDX = require('@next/mdx')()

const nextConfig = {
    images: {
      remotePatterns: [
        {
          protocol: 'https',
          hostname: 'logo.clearbit.com',
          port: '',
          pathname: '/**',
        },
      ],
    },
    pageExtensions: ['js', 'jsx', 'mdx', 'ts', 'tsx'],
    plugins: [
      require('@tailwindcss/typography'),
      // ...
    ],
  
  }
  

module.exports = withMDX(nextConfig)

