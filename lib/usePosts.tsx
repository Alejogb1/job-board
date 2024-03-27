import { useQuery } from '@tanstack/react-query'
import getFilteredPosts from './getFilteredPosts';

const fetchPosts = async (query:string) => {
    const postsData: Promise<any> = getFilteredPosts(1, query, ""); // Fetch data for the first page
    const posts: Post[] = await postsData;
    return posts
}

const usePosts = (input:string) => {
  return useQuery({
    queryKey: [input],
    queryFn: (query) => fetchPosts(query.queryKey[0]),
    refetchOnWindowFocus: false,
  })
}

export { usePosts, fetchPosts }