'use server'
import prisma from "./prisma"

export default async function getTag(tagId: number) {
    try {
        const tag: any[] = await prisma.tag.findMany({
            where: {
                id : tagId,
            },
        });

        return tag;
    } catch (error) {
        console.error('Database Error:', error);
        throw new Error('Failed to fetch job posts.');
    }
}
