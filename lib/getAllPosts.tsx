'use server'
import prisma from "./prisma"

export default async function getAllPosts() {

    const jobs = await prisma.jobPost.findMany({ take: 20})
    
    if(!jobs) throw new Error('failed to fetch data')
    return {
        jobs 
    }
}