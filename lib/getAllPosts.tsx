// lib/jobPosts.ts

import prisma from './prisma';

export async function getAllPosts() {
    try {
        // Fetch only the fields needed for the sitemap
        const jobs = await prisma.jobPost.findMany({
            select: {
                slug: true, // Adjust to the correct field name
            },
        });

        // Map jobs to a structure needed for the sitemap
        return jobs.map((job) => ({
            slug: job.slug,
            url: `https://www.jobseekr.ai/posts/${job.slug}`,
        }));
    } catch (error) {
        console.error('Database Error:', error);
        throw new Error('Failed to fetch job post slugs.');
    }
}
