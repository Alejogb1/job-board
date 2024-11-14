import {getAllPosts} from "./getAllPosts";

export default function fetcher() {
    const postsData = getAllPosts()
    return {
        postsData
    }
}