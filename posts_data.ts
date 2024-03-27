'use server'
import getFilteredPosts from "./lib/getFilteredPosts";
interface Post {
    id: number,
    post_by_id: number,
    is_active: boolean,
    is_remote: boolean,
    is_sponsored: boolean,
    job_title: string,
    job_body: string,
    slug: string,
    job_post_url: string,
    created_at: Date,
  }

const getData = async () => {
    const postsData: Promise<any> = getFilteredPosts(1,"data","texas"); // Fetch data for the first page
    const filteredData: Post[] = await postsData;
    return filteredData
}


console.log("what getData returns: ", getData())

export const POSTS = getData()