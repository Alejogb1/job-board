  
'use server'
import prisma from "./prisma"

export default async function getCompany(input:number) {
    const company = await prisma.company.findUnique(
        {where  : {
            id : input
        }}
    )
    console.log("company: ", company)
    if(!company) throw new Error('failed to fetch data')
    return {
        company 
    }
}