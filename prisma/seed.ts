import { PrismaClient } from "@prisma/client";
const prisma = new PrismaClient()

async function main() {
    const data = await prisma.jobPost.update({
      where: { company_id: id },
      connect: {
        company: [{id : id},],
      },
    })
  }
  main()
    .then(async () => {
      await prisma.$disconnect()
    })
    .catch(async (e) => {
      console.error(e)
      await prisma.$disconnect()
      process.exit(1)
    })
  