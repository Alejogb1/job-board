'use server'
import prisma from "./prisma"

export default async function getTagsOnPosts(postId: number) {
    try {
        const tags: any[] = await prisma.tagsOnPosts.findMany({
            where: {
                postId : postId,
            },
        });

        return tags;
    } catch (error) {
        console.error('Database Error:', error);
        throw new Error('Failed to fetch job posts.');
    }
}
