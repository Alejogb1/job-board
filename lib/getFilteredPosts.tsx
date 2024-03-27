'use server'
import prisma from "./prisma"

export default async function getFilteredPosts(page: number, term:string, location:string) {
    const itemsPerPage = 10;
    const skip = (page - 1) * itemsPerPage;
    try {
        if (term && location) {
            const posts: any[] = await prisma.jobPost.findMany({
                take: itemsPerPage,
                skip: skip,
                where: {
                    job_title: {
                        search: term
                    },
                    region: {
                        search: location
                    }
                }
            });
            return posts;
        } else if (term) {
            const posts: any[] = await prisma.jobPost.findMany({
                take: itemsPerPage,
                skip: skip,
                where: {
                    job_title: {
                        search: term
                    }
                }
            });
            return posts;
        } else if (location) {
            const posts: any[] = await prisma.jobPost.findMany({
                take: itemsPerPage,
                skip: skip,
                where: {
                    region: {
                        search: location
                    }
                }
            });
            return posts;
        }
        else {
            const posts: any[] = await prisma.jobPost.findMany({
                take: itemsPerPage,
                skip: skip,
            });
            return posts;
        }
    } catch (error) {
        console.log(error)
        console.error('Database Error:', error);
        throw new Error('Failed to fetch job posts.');
    }
}
