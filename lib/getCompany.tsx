import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export default async function getCompany(input: number) {
    if (!input) {
        throw new Error("Invalid input: Company ID is required.");
    }

    try {
        const company = await prisma.company.findUnique({
            where: {
                id: input,
            },
        });

        if (!company) {
            throw new Error(`No company found with ID: ${input}`);
        }

        return company;
    } finally {
        await prisma.$disconnect();
    }
}
