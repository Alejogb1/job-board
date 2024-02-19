'use server';
import { string } from "zod";
import prisma from "./prisma";
import { PrismaClient, Prisma } from '@prisma/client'

export default async function getFilteredPostsWithTags(page: number, term: string, location: string, tags:unknown[], remote:string) {
    const itemsPerPage = 10;
    const skip = (page - 1) * itemsPerPage;


    try {
        let whereClause: any = {}
        if (typeof tags == "string" && tags){
            let tag_id = 0
            switch (tags) {
                case 'fulltime':
                    tag_id = 370
                  break;
                case 'partTime':
                    tag_id = 371
                break;
                case 'intern':
                    tag_id = 366
                  break;
                case 'contract':
                    tag_id = 368
                  break;
                case 'senior':
                  tag_id = 362
                  break;
                case 'mid':
                  tag_id = 363
                  break;
                case 'entry':
                  tag_id = 366
                  break;
                case 'executive':
                    tag_id = 369	
                  break;
                default:
                  console.log(`out of ${tags}`);
              }
            whereClause = {
                tags: {
                    some: {
                        tagId : tag_id
                    }
                }
            };
    
        }
        if (typeof tags == "object") {
            tags.map((tag, index) => {
                switch (tag) {
                    case 'fulltime':
                        tags[index] = 370
                      break;
                    case 'partTime':
                        tags[index] = 371
                    break;
                    case 'intern':
                        tags[index] = 366
                      break;
                    case 'contract':
                        tags[index] = 368
                      break;
                    case 'senior':
                      tags[index] = 362
                      break;
                    case 'mid':
                      tags[index] = 363
                      break;
                    case 'entry':
                      tags[index] = 361
                      break;
                    case 'executive':
                      tags[index] = 369	
                      break;
                    default:
                      console.log(`out of ${tag}`);
                  }
            })
            whereClause = {
              OR: [
                {
                  tags: {
                    some: {
                      tagId: tags[0],
                    },
                  },
                },
                {
                  tags: {
                    some: {
                      tagId: tags[1],
                    },
                  },
                },
              ],

            }
        }

        if (term) {
            whereClause.job_title = { search: term };
        }

        if (location) {
            whereClause.region = { search: location };
        }
        if (remote) {
          whereClause.is_remote = true;
        }

        console.log("WHERE CLAUSE: ", remote)

        try {
          const posts: any[] = await prisma.jobPost.findMany({
            take: itemsPerPage,
            skip: skip,
            where: whereClause,
          });    

        return posts 
           
      } catch (e) {
        console.log("event prisma ", e)
          if (e instanceof Prisma.PrismaClientKnownRequestError) {
            console.log("event prisma ", e)
            // The .code property can be accessed in a type-safe manner
            if (e.code === 'P2002') {
              console.log(
                'There is a unique constraint violation, a new user cannot be created with this email'
              )
            }
          }
          throw e
        }
        
               
    } catch (error) {
        console.error('Database Error:', error);
        throw new Error('Failed to fetch job posts.');
    }
}
