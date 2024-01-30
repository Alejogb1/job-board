'use server'
import prisma from "./prisma"

export default async function getFilteredPosts(page: number) {
    const itemsPerPage = 10;
    const skip = (page - 1) * itemsPerPage;

    try {
        const posts: any[] = await prisma.jobPost.findMany({
            take: itemsPerPage,
            skip: skip,
        });

        return posts;
    } catch (error) {
        console.error('Database Error:', error);
        throw new Error('Failed to fetch job posts.');
    }
}