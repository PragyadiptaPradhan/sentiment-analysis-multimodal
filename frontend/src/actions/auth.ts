"use server";

import { hash } from "bcrypt";
import { signupSchema, type SignupSchema } from "~/schemas/auth";
import { db } from "~/server/db";
import crypto from "crypto";


export async function registerUser(data: SignupSchema) {
    try{
        //Server-side validation
        const result = signupSchema.safeParse(data);
        if(!result.success){
            return {error: "Invalid Data"};
        }

        const {name, email, password} = result.data;

        //Check if user already exists
        const existingUser = await db.user.findUnique({
            where: {email},

        });

        if (existingUser){
            return {error: "User already exists"};
        }

        const hashedPassword = await hash(password, 12);

        const user = await db.user.create({
            data: {
                name,
                email,
                password: hashedPassword,
                apiQuota: {
                    create: {
                        secretKey: `sa_live_${crypto.randomBytes(24).toString("hex")}`,
                    }
                },
            },
        });

        return {sucess: true};

    } catch(error){
        return { error: "Something went wrong"};
    }
}