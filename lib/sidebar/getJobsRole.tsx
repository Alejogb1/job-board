  
'use server'
import prisma from "../prisma"

export default async function getTagsPosts(input:any) {
        const postsTags = await prisma.tagsOnPosts.findMany({
            where : {
                AND : [
                    {
                        tagId : 470
                    },
                    {
                        tagId : 120
                    },
                    {
                        tagId : 12
                    },
                ]
            }
        })
        if(!postsTags) throw new Error('failed to fetch data')
        return {
            postsTags 
        }
}