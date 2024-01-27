export default function extractDomain(url:String) {
    // Remove protocol
    var domain = url.replace(/^(https?:\/\/)?(www\.)?/i, '');
    
    // Remove path and query parameters
    domain = domain.replace(/\/.*$/, '');

    return domain;
}