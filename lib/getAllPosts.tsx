'use server'
import prisma from "./prisma"

export default async function getAllPosts() {
    try {
        const jobs: any[] = await prisma.jobPost.findMany();
        return jobs;
    } catch (error) {
        console.error('Database Error:', error);
        throw new Error('Failed to fetch job posts.');
    }
}
