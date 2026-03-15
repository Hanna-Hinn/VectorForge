interface HeaderProps {
  title?: string;
}

export default function Header({ title }: HeaderProps) {
  return (
    <header className="flex h-14 items-center border-b border-gray-200/80 bg-white/80 px-6 backdrop-blur-sm dark:border-gray-700/80 dark:bg-gray-900/80">
      {title && (
        <h1 className="text-lg font-semibold tracking-tight text-gray-900 dark:text-gray-100">
          {title}
        </h1>
      )}
    </header>
  );
}
