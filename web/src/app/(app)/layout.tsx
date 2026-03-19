import { auth } from "@/auth";
import { redirect } from "next/navigation";
import AppSidebar from "@/components/app/AppSidebar";
import AppTopBar from "@/components/app/AppTopBar";
import styles from "./layout.module.css";

const allowDevGuest =
  process.env.NODE_ENV !== "production" &&
  process.env.ALLOW_LOCAL_DEV_PROXY_FALLBACK !== "0";

export default async function AppLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const session = await auth();
  if (!session && !allowDevGuest) redirect("/signin");

  return (
    <div className={styles.layout}>
      <AppSidebar user={session?.user} />
      <div className={styles.right}>
        <AppTopBar user={session?.user} />
        <main className={styles.main}>{children}</main>
      </div>
    </div>
  );
}
