'use server'
import prisma from "./prisma"

export default async function getFilteredPosts(page: number, term:string) {
  console.log("PAGE: ", page)
  try {
    const itemsPerPage = 10;
    let skip = (1 - 1) * itemsPerPage;

    if(page) {
      skip = (page - 1) * itemsPerPage;
    }

    let whereClause: any = {}
    if (term) {
      whereClause.job_title = { search: term };
    }
      console.log("WHERE CLAUSE: ", whereClause)
      try {
      const posts: any[] = await prisma.jobPost.findMany({
        take: itemsPerPage,
        skip: skip,
        where: whereClause,
      });    
  
      return posts 
    } catch (error) {
    console.log(error)
    console.error('Database Error:', error);
    throw new Error('Failed to fetch job posts.');
    }    
  } catch (error) {
    console.error('Database Error:', error);
    throw new Error('Failed to fetch job posts.');
  }
}