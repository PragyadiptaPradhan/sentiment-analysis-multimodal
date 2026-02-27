import { PutObjectCommand, S3Client } from "@aws-sdk/client-s3";
import { InvokeEndpoint$, InvokeEndpointCommand, SageMakerRuntimeClient } from "@aws-sdk/client-sagemaker-runtime";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { NextResponse } from "next/server";
import { checkAndUpdateQuota } from "~/lib/quota";
import { db } from "~/server/db";

export async function POST(req: Request) {
  try {
    //Get API key from authorization header
    const apiKey = req.headers.get("Authorization")?.replace("Bearer ", "");
    if (!apiKey) {
      return NextResponse.json({ error: "API key required." }, { status: 401 });
    }

    //Find the user by the API key
    const quota = await db.apiQuota.findFirst({
      where: {
        secretKey: apiKey,
      },
      select: {
        userId: true,
      },
    });

    if (!quota) {
      return NextResponse.json({ error: "Invalid API key." }, { status: 401 });
    }

    const { key } = await req.json();

    if (!key) {
      return NextResponse.json({ error: "Key is required." }, { status: 400 });
    }

    const file = await db.videoFile.findUnique({
      where: { key },
      select: { userId: true, analyzed: true },
    });

    if (!file) {
      return NextResponse.json({ error: "File not found." }, { status: 404 });
    }

    if (file.userId !== quota.userId) {
      return NextResponse.json({ error: "Unauthorized." }, { status: 403 });
    }

    if (file.analyzed) {
      return NextResponse.json(
        { error: "File has already been analyzed." },
        { status: 400 },
      );
    }

    const hasQuota = await checkAndUpdateQuota(quota.userId, true);

    if (!hasQuota) {
      return NextResponse.json({ error: "Quota exceeded." }, { status: 429 });
    }

    //Call sagemaker endpoint
    const sagemakerClient = new SageMakerRuntimeClient({
      region: process.env.AWS_REGION,
      credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
      },
    });

    const bucketName = process.env.AWS_INFERENCE_BUCKET!.startsWith("arn:aws:s3:::")
      ? process.env.AWS_INFERENCE_BUCKET!.split(":").pop()
      : process.env.AWS_INFERENCE_BUCKET!;

    const command = new InvokeEndpointCommand({
      EndpointName: process.env.AWS_ENDPOINT_NAME!,
      ContentType: "application/json",
      Body: JSON.stringify({
        video_path: `s3://${bucketName}/${key}`,
      }),
    });

    const response = await sagemakerClient.send(command);
    const analysis = JSON.parse(new TextDecoder().decode(response.Body));

    await db.videoFile.update({
      where: { key },
      data: {
        analyzed: true,
      },
    });



    return NextResponse.json({
      analysis,
    });
  } catch (error) {
    console.error("Analysi error: ", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 },
    );
  }
}
